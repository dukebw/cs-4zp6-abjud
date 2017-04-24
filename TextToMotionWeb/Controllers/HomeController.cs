using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using TextToMotionWeb.Data;
using Microsoft.AspNetCore.Authorization;
using TextToMotionWeb.Models;
using Microsoft.AspNetCore.Identity;
using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace TextToMotionWeb.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index()
        {
            var dbcontext = new ApplicationDbContext();
            dbcontext.Database.EnsureCreated();
            return View();
        }

        public IActionResult About()
        {
            return View();
        }

        public IActionResult Contact()
        {
            return View();
        }

        public IActionResult Error()
        {
            return View();
        }
    }
}
